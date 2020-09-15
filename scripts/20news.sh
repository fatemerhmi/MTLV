#download + unzip the file and remove the tar file after ward
wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
mkdir 20news-bydate
tar -xvf 20news-bydate.tar.gz -C {data_dir}
rm -r 20news-bydate.tar.gz
