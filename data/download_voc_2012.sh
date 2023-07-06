start=`date +%s`

# check if argument is provided
if [ -z "$1" ]; then
    # navigate to ~/data
    echo "provide an argument for the download directory"
    exit 0
fi

# check if is valid directory
if [ ! -d $1 ]; then
    echo $1 "is not a valid directory"
    exit 0
fi

echo "Navigating to" $1 "..."
cd $1

echo "Downloading VOC2012 trainval ..."
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
echo "Done downloading."

echo "Extracting trainval ..."
tar -xvf VOCtrainval_11-May-2012.tar
echo "Removing tar ..."
rm VOCtrainval_11-May-2012.tar

echo "Moving all content in VOCdevkit/VOC2012 to" $1 "..."
mv ./VOCdevkit/VOC2012/* ./
rm -rf ./VOCdevkit

end=`date +%s`
runtime=$((end-start))
echo "Completed in" $runtime "seconds"
