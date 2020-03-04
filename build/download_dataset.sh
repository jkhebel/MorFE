# Get data directory (prompt if not passed)
if [[ $# -eq 0 ]] ; then
    echo 'Enter the directory path where you would like to store the data:  '
    read DATA_DIR
else
    DATA_DIR=$1
fi

# Store images in subdir
IMG_DIR="${DATA_DIR}/images"

# Download metadata
wget https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_image.csv -P $DATA_DIR
# Download image archives
wget -i build/BBBC022_v1_images_urls.txt -P $IMG_DIR

# Unzip image archives
unzip "${IMG_DIR}/*.zip" -d $IMG_DIR
# Delete zip files
rm -r $IMG_DIR/*.zip
