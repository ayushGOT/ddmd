#!/bin/bash

for md in /project/zerze/ayushg/argonne/ddmd/rna2/test/md_run/md_run*
do
cd "$md"

plumed driver --mf_dcd ./output.dcd --plumed /project/zerze/ayushg/argonne/ddmd/analysis/plumed_driver_Q.dat --pdb /project/zerze/ayushg/argonne/ddmd/analysis/gaga_native.pdb
done
