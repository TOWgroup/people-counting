# retinaface_deepstream
#create file  .so
cd nvdsinfer_customparser/
make all
cd ..
#Run deepstream-app
deepstream-app -c retina_network_config.txt
#Note: change path video and path enginefile
