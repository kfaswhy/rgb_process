#include "isp_gain.h"
//#include "rgb_process.h"
U8 isp_gain_process(RGB* img, IMG_CONTEXT context, G_CONFIG cfg)
{
    if (cfg.isp_gain_on == 0)
    {
        return OK;
    }
    

    for (int i = 0; i < context.full_size; i++)
    {
        img[i].r = clp_range(0, ((U16)img[i].r * cfg.isp_gain) >> 10, U8MAX);
        img[i].g = clp_range(0, ((U16)img[i].g * cfg.isp_gain) >> 10, U8MAX);
        img[i].b = clp_range(0, ((U16)img[i].b * cfg.isp_gain) >> 10, U8MAX);

    }
    save_img_with_timestamp(img, &context, "_ispgain");
    LOG("done.");
    return OK;
}