CC=gcc
CXX=g++
CPPFLAGS= -Wall -g -c -fPIC -std=c++14 -I./include

OBJS =   ./mobilenetv2.o  

TARGET = ./mobilenetv2

./mobilenetv2 : ./obj/mobilenetv2.o 
	g++ -Wall -g -std=c++11 -I./include -o ./mobilenetv2 $(OBJS)

./obj/mobilenetv2.o : ./mobilenetv2.cpp
	$(CXX) $(CPPFLAGS) -o ./mobilenetv2.o mobilenetv2.cpp

clean :
	rm -rf $(OBJS)


