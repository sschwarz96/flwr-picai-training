#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9, min_lr=1e-6):
    """
    Polynomial decay from initial_lr → min_lr over max_epochs.
    epoch is 1-indexed here.
    """
    decay = (1 - epoch / max_epochs) ** exponent
    return min_lr + (initial_lr - min_lr) * decay