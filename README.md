link to group project : https://github.com/Augmentales



first you should start your server (server.mjs). then run this. 

sudo apt update

sudo apt install build-essential cmake

sudo apt install libopencv-dev

sudo apt install libboost-all-dev

sudo apt install libwebsocketpp-dev nlohmann-json3-dev

mkdir build && cd build

cmake ..

make

./my_app
