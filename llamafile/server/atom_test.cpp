// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "atom.h"
#include "image.h"
#include <cstdlib>

namespace lf {
namespace server {
namespace {

const Atom atoms[] = {
    Atom(),
    Atom(-1),
    Atom(0),
    Atom(1),
    Atom(new Image("hello", 1)),
    Atom(new Image("there", 1)),
};

const size_t n = sizeof(atoms) / sizeof(atoms[0]);

void
test_atom_operator_lt()
{
    // irreflexivity: no element can be less than itself
    for (size_t i = 0; i < n; ++i)
        if (atoms[i] < atoms[i])
            exit(1);

    // asymmetry: if x < y then !(y < x)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (atoms[i] < atoms[j] && atoms[j] < atoms[i])
                exit(2);

    // transitivity: If x < y and y < z then x < z
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k)
                if (atoms[i] < atoms[j] && atoms[j] < atoms[k])
                    if (!(atoms[i] < atoms[k]))
                        exit(3);

    // transitivity of incomparability
    // if x ≈ y (neither < the other) and y ≈ z, then x ≈ z
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                bool i_equiv_j =
                  !((atoms[i] < atoms[j]) || (atoms[j] < atoms[i]));
                bool j_equiv_k =
                  !((atoms[j] < atoms[k]) || (atoms[k] < atoms[j]));
                bool i_equiv_k =
                  !((atoms[i] < atoms[k]) || (atoms[k] < atoms[i]));
                if (i_equiv_j && j_equiv_k && !i_equiv_k)
                    exit(4);
            }
        }
    }
}

void
test_atom_operator_eq()
{
    // reflexivity: every element must equal itself
    for (size_t i = 0; i < n; ++i)
        if (!(atoms[i] == atoms[i]))
            exit(5);

    // symmetry: if x == y then y == x
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if ((atoms[i] == atoms[j]) != (atoms[j] == atoms[i]))
                exit(6);

    // transitivity: if x == y and y == z then x == z
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k)
                if (atoms[i] == atoms[j] && atoms[j] == atoms[k])
                    if (!(atoms[i] == atoms[k]))
                        exit(7);

    // consistency with operator<: if x == y then !(x < y) && !(y < x)
    // this ensures == and < agree on their notion of equivalence
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (atoms[i] == atoms[j])
                if (atoms[i] < atoms[j] || atoms[j] < atoms[i])
                    exit(8);
}

void
atom_test()
{
    test_atom_operator_lt();
    test_atom_operator_eq();
}

} // namespace
} // namespace server
} // namespace lf

int
main()
{
    lf::server::atom_test();
}
