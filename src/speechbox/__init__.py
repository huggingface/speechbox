# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

__version__ = "0.2.0"

from .utils import (is_accelerate_available, is_pyannote_available,
                    is_scipy_available, is_torchaudio_available,
                    is_transformers_available)

if is_transformers_available() and is_accelerate_available() and is_scipy_available():
    from .restore import PunctuationRestorer
else:
    from .utils.dummy_transformers_and_accelerate_and_scipy_objects import *

if is_transformers_available() and is_torchaudio_available() and is_pyannote_available():
    from .diarize import ASRDiarizationPipeline
else:
    from .utils.dummy_transformers_and_torchaudio_and_pyannote_objects import *
