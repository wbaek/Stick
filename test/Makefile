include ../src/Makefile.include

GTEST_PATH=$(DEPENDENCY_PATH)/gtest/
INCLUDE += -isystem $(GTEST_PATH)/include/ \
		   -I./

STATIC_LIBS +=\
			  $(GTEST_PATH)/lib/.libs/libgtest.a \
			  ../src/lib$(PRODUCT_NAME).a 

CC_SRCS=$(call rwildcard,,*.cpp)
CC_OBJS=$(CC_SRCS:.cpp=.o)

OBJS=$(CC_OBJS)

TARGET=unit_test_main.e

$(TARGET): $(OBJS)
	$(GCC) $(LDFLAGS) $(OBJS) $(LIB_PATH) $(DYNAMIC_LIBS) $(STATIC_LIBS) -o $@

build: $(TARGET)

test: run
run: build
	./$(TARGET)
	@printf "\033[0;32m============================================================\033[0m\n"
	@printf "\033[0;32m= TEST Complete : $(PROJECT_NAME) \033[0m\n"
	@printf "\033[0;32m============================================================\033[0m\n"

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)
