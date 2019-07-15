#!/bin/bash
#@author: hugonnet
#bash script to retrieve aster l1a recursively from a csv file of HTTPS directories

#csv list of LPDAAC PullDirs: dos2unix seems to be necessary quite often here to read the csv
input=$1 #if end of string displays with Windows carrier (i.e %OD, resulting in Error 404 for wget: correct csv file with: dos2unix(list.csv))
#download folder output
output=$2
#number of simultaneous downloads
nb_paral=10

cd ~
touch .netrc
echo "machine urs.earthdata.nasa.gov login <earthdata_username> password <earthdata_password>" >> .netrc
chmod 0600 .netrc
cd ~
touch .urs_cookies

# download granules zip and met recursively
cat $input | parallel --verbose --delay 1 -j $nb_paral --results $output wget --no-check-certificate --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -np -l 1 -A *.zip,*.met {} -P $output

# download zip archive of ~100 granules >> easier to check for errors
# it would be easier... if only LPDAAC wasn't zipping files over 4Go on an old Unix system:
# data gets corrupted and become unusable: download image by image instead
#cat $input | parallel --verbose --delay 1 -j $nb_paral --results $output wget --no-check-certificate --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies {} -P $output
