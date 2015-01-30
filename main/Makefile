include ../src/Makefile.include

CC_SRCS=$(call rwildcard,,*.cpp)
CC_OBJS=$(CC_SRCS:.cpp=.o)

OBJS=$(CC_OBJS)

STATIC_LIBS := ../src/lib$(PRODUCT_NAME).a $(STATIC_LIBS)

TARGET=$(OBJS:.o=)
build:$(TARGET)

$(TARGET):$(OBJS)
	$(GCC) $(LDFLAGS) $@.o $(LIB_PATH) $(DYNAMIC_LIBS) $(STATIC_LIBS) -o $@.e
	@printf "\033[0;32m============================================================\033[0m\n"
	@printf "\033[0;32m= Build Complete : $@ \033[0m\n"
	@printf "\033[0;32m============================================================\033[0m\n"

clean:
	rm -rf *.e
	rm -f $(OBJS)
	rm -rf *.dSYM || true # for mac

install:
	mkdir -p ../bin
	cp *.e ../bin
	rm -f ../bin/$(PROJECT_NAME).e
	ln -s $(PRODUCT_NAME).e ../bin/$(PROJECT_NAME).e
	@printf "\033[0;32m============================================================\033[0m\n"
	@printf "\033[0;32m= Install Complete : $(TARGET) \033[0m\n"
	@printf "\033[0;32m============================================================\033[0m\n"
