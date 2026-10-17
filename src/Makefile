TARGET = KurierKhm
OBJS = main.o assets_data.o
CFLAGS = -O2 -G0 -Wall -Wno-misleading-indentation
LIBS = -lz -lm
EXTRA_TARGETS = EBOOT.PBP
PSP_EBOOT_TITLE = KURIER KHMELNYTSKYI
PSP_EBOOT_ICON = ICON0.PNG
PSPSDK = $(shell psp-config --pspsdk-path)
include $(PSPSDK)/lib/build.mak
main.o: main.c assets.h pal_ids.h
