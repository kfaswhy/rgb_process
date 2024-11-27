#include "ob.h"

U8 awb_process(RGB *rgb, IMG_CONTEXT context, G_CONFIG cfg)
{
    if (cfg.awb_on == 0)
    {
        return OK;
    }

    RGB* p_rgb = &rgb[0];
    U16 tmp = 0;

    for (int i = 0; i < context.full_size; i++)
    {
        tmp = p_rgb->b;
        tmp = (U16)p_rgb->b * cfg.b_gain;
        tmp = (U16)p_rgb->b * cfg.b_gain + 512;
        tmp = ((U16)p_rgb->b * cfg.b_gain + 512)>>10;



        p_rgb->r = clp_range(0, ((U16)p_rgb->r * cfg.r_gain + 512) >> 10, U8MAX);
        tmp = ((U16)p_rgb->b * cfg.b_gain+512)>>10;
        p_rgb->b = clp_range(0, ((U16)p_rgb->b * cfg.b_gain + 512) >> 10, U8MAX);

        p_rgb++;
    }

    LOG("done.");
    return OK;
}