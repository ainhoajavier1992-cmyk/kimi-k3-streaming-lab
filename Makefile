CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?= -lm

BUILD_DIR := build
BIN := $(BUILD_DIR)/k3stream
SRC := src/k3stream.c

.PHONY: all clean smoke probe test

all: $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN): $(SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

smoke: $(BIN)
	rm -rf fixtures/tiny
	mkdir -p fixtures
	$(BIN) fixture --out fixtures/tiny --layers 4 --experts 16 --hidden 32 --inter 64 --topk 2 --seed 7
	$(BIN) inspect --model fixtures/tiny
	$(BIN) run --model fixtures/tiny --tokens 12 --cache 3 --trace

test: $(BIN)
	python3 tools/run_tests.py

probe:
	python3 tools/kimi_probe.py

clean:
	rm -rf $(BUILD_DIR) fixtures/tiny
