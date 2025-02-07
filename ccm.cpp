#include "ccm.h"
//#include "rgb_process.h"
U8 ccm_process(RGB* rgb, IMG_CONTEXT context, G_CONFIG cfg)
{
    if (cfg.ccm_on == 0)
    {
        return OK;
    }
    
    RGB* p_rgb = &rgb[0];
    float tmpr = 0.0;
    float tmpg = 0.0;
    float tmpb = 0.0;
    for (int i = 0; i < context.full_size; i++)
    {

        // Ӧ��ɫ��У������
        tmpr = (float)cfg.ccm[0] * p_rgb->r + cfg.ccm[1] * p_rgb->g + cfg.ccm[2] * p_rgb->b;
        tmpg = (float)cfg.ccm[3] * p_rgb->r + cfg.ccm[4] * p_rgb->g + cfg.ccm[5] * p_rgb->b;
        tmpb = (float)cfg.ccm[6] * p_rgb->r + cfg.ccm[7] * p_rgb->g + cfg.ccm[8] * p_rgb->b;

        p_rgb->r = clp_range(0, (tmpr + 0.5), 255);
        p_rgb->g = clp_range(0, (tmpg + 0.5), 255);
        p_rgb->b = clp_range(0, (tmpb + 0.5), 255);

        p_rgb++;
    }

    save_img_with_timestamp(rgb, &context, "_ccm");

    LOG("done.");

    return OK;
}