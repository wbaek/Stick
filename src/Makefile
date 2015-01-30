include Makefile.include

CC_SRCS=$(call rwildcard,,*.cpp)
CC_OBJS=$(CC_SRCS:.cpp=.o)

OBJS=$(CC_OBJS)

TARGET=lib$(PRODUCT_NAME).a

$(TARGET): print_environments $(OBJS)
	@ar -crs $@ $(OBJS)
	@printf "\033[0;32m============================================================\033[0m\n"
	@printf "\033[0;32m=  Build  Complete : $(TARGET) \033[0m\n"
	@printf "\033[0;32m============================================================\033[0m\n"

build: $(TARGET)

install: build 
	mkdir -p ../lib
	cp -f $(TARGET) ../lib/
	rm -f ../lib/lib$(PROJECT_NAME).a
	ln -s lib$(PRODUCT_NAME).a ../lib/lib$(PROJECT_NAME).a
	@printf "\033[0;32m============================================================\033[0m\n"
	@printf "\033[0;32m= Install Complete : lib/$(TARGET) \033[0m\n"
	@printf "\033[0;32m============================================================\033[0m\n"

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)

