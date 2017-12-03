OBJS=

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
	CC=/usr/bin/clang
	CFLAGS= -std=c11 -g -Wall -I/usr/lib/llvm-6.0/include
	CXX=/usr/bin/clang++
	CXXFLAGS= -std=c++17 -g -Wall -O2 -I/usr/lib/llvm-6.0/include
	# LDFLAGS= -L/usr/lib/llvm-6.0/lib
	LDFLAGS= -L/usr/lib/llvm-6.0/lib -Wl,-rpath,/usr/lib/llvm-6.0/lib
endif
ifeq ($(UNAME_S),Darwin)
	CC=/usr/local/opt/llvm/bin/clang
	CFLAGS= -std=c11 -g -Wall -I/usr/local/opt/llvm/include
	CXX=/usr/local/opt/llvm/bin/clang++
	CXXFLAGS= -std=c++14 -g -Wall -O2 -I/usr/local/opt/llvm/include
	# LDFLAGS= -L/usr/local/opt/llvm/lib
	LDFLAGS= -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib
endif

LDLIBS=

RM=rm -f

$(P): $(OBJS)

all:	clear $(P) t

t:
	./$(P)

clean:
	$(RM) $(P)

clear:
	@clear
