#include <ctime>
#include <iomanip>
#include <sstream>
#include "degamma.h"

U8 degamma_process(RGB* rgb, IMG_CONTEXT context, G_CONFIG cfg)
{
    if (cfg.degamma_on == 0)
    {
        return OK;
    } 
    U32 tmp_x = 0;
    U32 tmp_y = 0;
    RGB* p_rgb = &rgb[0];

    for (int i = 0; i < context.full_size; i++)
    {
        tmp_x = (U32)p_rgb->r << 4;
        tmp_y = calc_inter(tmp_x, cfg.degamma_x, cfg.degamma_y, 49);
        p_rgb->r = clp_range(0, (tmp_y + 8) >> 4, U8MAX);

        tmp_x = (U32)p_rgb->g << 4;
        tmp_y = calc_inter(tmp_x, cfg.degamma_x, cfg.degamma_y, 49);
        p_rgb->g = clp_range(0, (tmp_y + 8) >> 4, U8MAX);

        tmp_x = (U32)p_rgb->b << 4;
        tmp_y = calc_inter(tmp_x, cfg.degamma_x, cfg.degamma_y, 49);
        p_rgb->b = clp_range(0, (tmp_y + 8) >> 4, U8MAX);

        p_rgb++;
    }

    save_img_with_timestamp(rgb, &context, "_degamma");

    LOG("done.");
    return OK;
}