#!/bin/bash
# link individual dems, correlation images, hillshades, and orthophotos from MMASTER into
# one directory for easy browsing/loading into QGIS.
hdir=$(pwd)
darr=$(ls -d AST_L1A*);

mkdir -p dems hillshades orthos corr
cd dems/
for d in ${darr[@]}; do 
  ln -s ../$d/$d\_Zadj3.tif $d\_Zadj3.tif;
done
cd ../

cd orthos
for d in ${darr[@]}; do
  ln -s ../$d/$d\_V123_adj1.tif $d\_V123_adj1.tif;
done
cd ..

cd hillshades
for d in ${darr[@]}; do
  ln -s ../$d/$d\_HS.tif $d\_HS.tif;
done
cd ..

cd corr
for d in ${darr[@]}; do
  ln -s ../$d/$d\_CORR_adj1.tif $d\_CORR_adj1.tif;
done
cd ..
