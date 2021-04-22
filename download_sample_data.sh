####################################################################
# Script to download and install the Pythie sample data
# needed to run the example notebooks
####################################################################

conda activate pythie
zenodo_get 10.5281/zenodo.4707154
unzip europa_grid_data.zip
unzip soltau_station_data.zip
mv LICENSE ./data/
mv md5sums.txt ./data/
rm -f europa_grid_data.zip soltau_station_data.zip

