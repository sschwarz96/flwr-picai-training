import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)

from src.picai_baseline.flwr.federated_training_methods import load_model_checkpoint
from src.picai_baseline.flwr.run_config import run_configuration, RunConfig
from src.picai_baseline.unet.training_setup.default_hyperparam import \
    get_default_hyperparams
from src.picai_baseline.unet.training_setup.neural_network_selector import \
    neural_network_for_run
from src.picai_baseline.unet.training_setup.preprocess_utils import z_score_norm
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample, PreprocessingSettings, crop_or_pad, resample_img
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter


class csPCaAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained baseline U-Net model from
    https://github.com/DIAGNijmegen/picai_baseline as a
    grand-challenge.org algorithm.
    """

    def __init__(self, weights_path: Path, input_dir: Path):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # directory to model weights
        self.weights_path = weights_path

        modalities = ["_t2w", "_adc", "_hbv"]
        modality_order = {modality: i for i, modality in enumerate(modalities)}

        def get_modality_index(file_name: Path):
            for modality in modalities:
                if modality in file_name:
                    return modality_order[modality]
            return len(modalities)  # If no match, push it to the end

        self.input_dir = input_dir
        self.image_input_paths = sorted(
            [file.name for file in self.input_dir.iterdir() if any(modality in file.name for modality in modalities)],
            key=get_modality_index
        )
        self.image_input_paths = [self.input_dir / path for path in self.image_input_paths]

        # load clinical information
        # with open("/input/clinical-information-prostate-mri.json") as fp:
        #     self.clinical_info = json.load(fp)

        # path to output files
        self.detection_map_output_path = Path(
            "/home/zimon/flwr-picai-training/workdir/output/images/cspca-detection-map/cspca_detection_map.mha")
        self.case_level_likelihood_output_file = Path(
            "/home/zimon/flwr-picai-training/workdir/output/cspca-case-level-likelihood.json")

        # create output directory
        self.detection_map_output_path.parent.mkdir(parents=True, exist_ok=True)

        # define compute used for training/inference ('cpu' or 'cuda')
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = 'cpu'

        # # extract available clinical metadata (not used for this example)
        # self.age = self.clinical_info["patient_age"]

        # if "PSA_report" in self.clinical_info:
        #     self.psa = self.clinical_info["PSA_report"]
        # else:
        #     self.psa = None  # value is missing, if not reported
        #
        # if "PSAD_report" in self.clinical_info:
        #     self.psad = self.clinical_info["PSAD_report"]
        # else:
        #     self.psad = None  # value is missing, if not reported
        #
        # if "prostate_volume_report" in self.clinical_info:
        #     self.prostate_volume = self.clinical_info["prostate_volume_report"]
        # else:
        #     self.prostate_volume = None  # value is missing, if not reported
        #
        # # extract available acquisition metadata (not used for this example)
        # self.scanner_manufacturer = self.clinical_info["scanner_manufacturer"]
        # self.scanner_model_name = self.clinical_info["scanner_model_name"]
        # self.diffusion_high_bvalue = self.clinical_info["diffusion_high_bvalue"]

        # define input data specs [image shape, spatial res, num channels, num classes]
        self.img_spec = {
            'image_shape': [20, 256, 256],
            'spacing': [3.0, 0.5, 0.5],
            'num_channels': 3,
            'num_classes': 2,
        }

        # load trained algorithm architecture + weights
        self.models = []
        model_arch = ['unet']
        model_folds = [range(5)]

        run_configuration = RunConfig()
        # init_parameters = load_model_checkpoint(neural_network_for_run(args=run_configuration))

        model = neural_network_for_run(args=run_configuration, device=self.device)

        # load trained weights for the fold
        checkpoint = torch.load(self.weights_path)
        print(f"Loading weights from {self.weights_path}")
        model.load_state_dict(checkpoint)
        model.to(self.device)
        self.models += [model]
        print("Complete.")
        print("-" * 100)

        # display error/success message
        if len(self.models) == 0:
            raise Exception("No models have been found/initialized.")
        else:
            print(f"Success! {len(self.models)} model(s) have been initialized.")
            print("-" * 100)

    # generate + save predictions, given images
    def predict(self):

        print("Preprocessing Images ...")

        # read images (axial sequences used for this example only)
        sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in self.image_input_paths
            ],
            settings=PreprocessingSettings(
                matrix_size=self.img_spec['image_shape'],
                spacing=self.img_spec['spacing']
            )
        )

        # preprocess - align, center-crop, resample
        sample.preprocess()
        cropped_img = [
            sitk.GetArrayFromImage(x)
            for x in sample.scans
        ]

        # preprocessing - intensity normalization + expand channel dim
        preproc_img = [
            z_score_norm(np.expand_dims(x, axis=0), percentile=99.5)
            for x in cropped_img
        ]

        # preprocessing - concatenate + expand batch dim
        preproc_img = np.expand_dims(np.vstack(preproc_img), axis=0)

        # preprocessing - convert to PyTorch tensor
        preproc_img = torch.from_numpy(preproc_img)

        print("Complete.")

        # test-time augmentation (horizontal flipping)
        img_for_pred = [preproc_img.to(self.device)]
        img_for_pred += [torch.flip(preproc_img, [4]).to(self.device)]

        # begin inference
        outputs = []
        print("Generating Predictions ...")

        # for each member model in ensemble
        for p in range(len(self.models)):
            # switch model to evaluation mode
            self.models[p].eval()

            # scope to disable gradient updates
            with torch.no_grad():
                # aggregate predictions for all tta samples
                preds = [
                    torch.sigmoid(self.models[p](x))[:, 1, ...].detach().cpu().numpy()
                    for x in img_for_pred
                ]

                # revert horizontally flipped tta image
                preds[1] = np.flip(preds[1], [3])

                # gaussian blur to counteract checkerboard artifacts in
                # predictions from the use of transposed conv. in the U-Net
                outputs += [
                    np.mean([
                        gaussian_filter(x, sigma=1.5)
                        for x in preds
                    ], axis=0)[0]
                ]

        # ensemble softmax predictions
        ensemble_output = np.mean(outputs, axis=0).astype('float32')

        # read and resample images (used for reverting predictions only)
        sitk_img = [
            sitk.ReadImage(str(path)) for path in self.image_input_paths
        ]
        resamp_img = [
            sitk.GetArrayFromImage(
                resample_img(x, out_spacing=self.img_spec['spacing'])
            )
            for x in sitk_img
        ]

        # revert softmax prediction to original t2w - reverse center crop
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(crop_or_pad(
            ensemble_output, size=resamp_img[0].shape))
        cspca_det_map_sitk.SetSpacing(list(reversed(self.img_spec['spacing'])))

        # revert softmax prediction to original t2w - reverse resampling
        cspca_det_map_sitk = resample_img(cspca_det_map_sitk,
                                          out_spacing=list(reversed(sitk_img[0].GetSpacing())))

        # process softmax prediction to detection map
        cspca_det_map_npy = extract_lesion_candidates(
            sitk.GetArrayFromImage(cspca_det_map_sitk), threshold='dynamic')[0]

        # remove (some) secondary concentric/ring detections
        cspca_det_map_npy[cspca_det_map_npy < (np.max(cspca_det_map_npy) / 5)] = 0

        # make sure that expected shape was matched after reverse resampling (can deviate due to rounding errors)
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(crop_or_pad(
            cspca_det_map_npy, size=sitk.GetArrayFromImage(sitk_img[0]).shape))

        # works only if the expected shape matches
        cspca_det_map_sitk.CopyInformation(sitk_img[0])

        # save detection map
        atomic_image_write(cspca_det_map_sitk, self.detection_map_output_path)

        # save case-level likelihood
        with open(str(self.case_level_likelihood_output_file), 'w') as f:
            json.dump(float(np.max(cspca_det_map_npy)), f)


if __name__ == "__main__":
    (csPCaAlgorithm(
        Path(
            "/home/zimon/flwr-picai-training/src/picai_baseline/flwr/outputs/2025-02-23/15-31-47/model_state_rank_0.6518681856663044_round_20.pth"),
        Path("/home/zimon/flwr-picai-training/input/images/10005"))
     .predict())
