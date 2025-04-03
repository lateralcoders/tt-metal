// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"

void kernel_main() { DPRINT << "Hello from my_writer kernel!" << ENDL(); }
