#!/bin/bash
while read file_name file_url
do
	wget -O ${file_name} -c ${file_url}
done < nuscenes.txt

tar -xvzf v1.0-trainval_meta.tgz
tar -xvzf v1.0-trainval01_keyframes.tgz
tar -xvzf v1.0-trainval02_keyframes.tgz
tar -xvzf v1.0-trainval03_keyframes.tgz
tar -xvzf v1.0-trainval04_keyframes.tgz
tar -xvzf v1.0-trainval05_keyframes.tgz
tar -xvzf v1.0-trainval06_keyframes.tgz
tar -xvzf v1.0-trainval07_keyframes.tgz
tar -xvzf v1.0-trainval08_keyframes.tgz
tar -xvzf v1.0-trainval09_keyframes.tgz
tar -xvzf v1.0-trainval10_keyframes.tgz
