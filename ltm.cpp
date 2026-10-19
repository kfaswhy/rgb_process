#include "ltm.h"

#define U16MAX 65535

U8 ltm_process(YUV* yuv, IMG_CONTEXT context, G_CONFIG cfg)
{
    if (cfg.ltm_on == 0) {
        return OK;
    }


    LOG("done.");
    return OK;
}