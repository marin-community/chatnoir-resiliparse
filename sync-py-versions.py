#!/usr/bin/env python3
# Copyright 2025 Janek Bevendorff
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

# Keep Python package versions in sync with Rust workspace version

import tomllib
import re
from subprocess import run

cargo_ver = tomllib.load(open('Cargo.toml', 'rb'))['workspace']['package']['version']

print(f'Updating Python package versions to {cargo_ver}.')

run(['poetry', 'version', '--project=fastwarc-py', cargo_ver], check=True)
run(['poetry', 'version', '--project=resiliparse-py', cargo_ver], check=True)

# Patch fastwarc dependency (cannot use `poetry add` for this if fastwarc is not yet published)
with open('resiliparse-py/pyproject.toml', 'r') as f:
    resiliparse_toml = f.read()
with open('resiliparse-py/pyproject.toml', 'w') as f:
    f.write(re.sub(r'"fastwarc==\d+\.\d+\.\d+"',
                   f'"fastwarc=={cargo_ver}"',
                   resiliparse_toml))
