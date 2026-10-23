TARGET = KurierKhm
OBJS = main.o gfx.o assets_data.o
CFLAGS = -O2 -G0 -Wall -Wno-misleading-indentation
LIBS = -lpspgu -lz -lm
EXTRA_TARGETS = EBOOT.PBP
PSP_EBOOT_TITLE = PROSKURIV EXPRESS
PSP_EBOOT_ICON = ICON0.PNG
PSPSDK = $(shell psp-config --pspsdk-path)
include $(PSPSDK)/lib/build.mak
main.o: main.c assets.h pal_ids.h gfx.h
gfx.o: gfx.c gfx.h
