#include "ltm.h"

#define U16MAX 65535
U8 ltm_process(U16* raw, IMG_CONTEXT context, G_CONFIG cfg)
{
    if (cfg.ltm_on == 0) {
        return OK;
    }

    U16 height = context.height;
    U16 width = context.width;
    U8 vblk = cfg.ltm_vblk;
    U8 hblk = cfg.ltm_hblk;

    if (vblk == 0 || hblk == 0) {
        return OK; // ������Ч��ֱ�ӷ���
    }

    // ÿ����ĳߴ�
    U16 blk_h = height / vblk;
    U16 blk_w = width / hblk;

    // ��ʼ��ֱ��ͼ������ӳ���
    U32* histograms[4]; // 4��ͨ���ֱ�ͳ��ֱ��ͼ
    float* ltm_curves[4]; // ����ӳ������

    for (int i = 0; i < 4; ++i) {
        histograms[i] = (U32*)calloc(U16MAX + 1, sizeof(U32));
        ltm_curves[i] = (float*)malloc((U16MAX + 1) * sizeof(float));
    }

    // �ֿ鴦��ͼ��
    for (U8 v = 0; v < vblk; ++v) {
        for (U8 h = 0; h < hblk; ++h) {
            // ��ǰ�����ʼ����
            U16 blk_start_y = v * blk_h;
            U16 blk_start_x = h * blk_w;

            // ��ǰ����ֱ��ͼͳ��
            memset(histograms[0], 0, (U16MAX + 1) * sizeof(U32));
            memset(histograms[1], 0, (U16MAX + 1) * sizeof(U32));
            memset(histograms[2], 0, (U16MAX + 1) * sizeof(U32));
            memset(histograms[3], 0, (U16MAX + 1) * sizeof(U32));

            U32 sum[4] = { 0 }; // ��ͨ�������ܺ�
            U16 max_brightness[4] = { 0 }; // ��ͨ���������

            for (U16 y = blk_start_y; y < blk_start_y + blk_h; ++y) {
                for (U16 x = blk_start_x; x < blk_start_x + blk_w; ++x) {
                    // ���� Bayer ��ʽѡ��ͨ��
                    int channel;
                    if (cfg.pattern == RGGB) {
                        channel = ((y % 2) << 1) + (x % 2);
                    }
                    else if (cfg.pattern == GRBG) {
                        channel = ((y % 2) << 1) + !(x % 2);
                    }
                    else if (cfg.pattern == GBRG) {
                        channel = (!(y % 2) << 1) + (x % 2);
                    }
                    else { // BGGR
                        channel = (!(y % 2) << 1) + !(x % 2);
                    }

                    U16 pixel_value = raw[y * width + x];
                    histograms[channel][pixel_value]++;
                    sum[channel] += pixel_value;
                    if (pixel_value > max_brightness[channel]) {
                        max_brightness[channel] = pixel_value;
                    }
                }
            }

            // �����ͨ����ƽ�����ȺͶԱȶ�������ֵ
            for (int c = 0; c < 4; ++c) {
                float avg_brightness = (float)sum[c] / (blk_h * blk_w / 4);
                float contrast_threshold = ((max_brightness[c] - avg_brightness) * cfg.ltm_cst_thdr + avg_brightness);

                // �Աȶ����ƣ�����ֱ��ͼ
                U32 excess = 0; // ���������ܺ�
                for (int i = 0; i <= U16MAX; ++i) {
                    if (histograms[c][i] > contrast_threshold) {
                        excess += histograms[c][i] - contrast_threshold;
                        histograms[c][i] = contrast_threshold;
                    }
                }

                // �������Ĳ��־��ȷ���
                U32 increment = excess / (U16MAX + 1);
                for (int i = 0; i <= U16MAX; ++i) {
                    histograms[c][i] += increment;
                }

                // ��������ӳ������
                U32 cumulative = 0;
                for (int i = 0; i <= U16MAX; ++i) {
                    cumulative += histograms[c][i];
                    ltm_curves[c][i] = (float)cumulative / (blk_h * blk_w / 4) * U16MAX;
                }
            }
        }
    }

    // ��ͼ��ÿ������Ӧ������ӳ�䣨˫���Բ�ֵ��
    for (U16 y = 0; y < height; ++y) {
        for (U16 x = 0; x < width; ++x) {
            int channel;
            if (cfg.pattern == RGGB) {
                channel = ((y % 2) << 1) + (x % 2);
            }
            else if (cfg.pattern == GRBG) {
                channel = ((y % 2) << 1) + !(x % 2);
            }
            else if (cfg.pattern == GBRG) {
                channel = (!(y % 2) << 1) + (x % 2);
            }
            else { // BGGR
                channel = (!(y % 2) << 1) + !(x % 2);
            }

            // ��ǰ�������ڿ��λ��
            float fy = (float)y / blk_h - 0.5f;
            float fx = (float)x / blk_w - 0.5f;

            int blk_y = (int)floor(fy);
            int blk_x = (int)floor(fx);
            float dy = fy - blk_y;
            float dx = fx - blk_x;

            blk_y = blk_y < 0 ? 0 : (blk_y >= vblk - 1 ? vblk - 1 : blk_y);
            blk_x = blk_x < 0 ? 0 : (blk_x >= hblk - 1 ? hblk - 1 : blk_x);

            // ˫���Բ�ֵ�ĸ����ӳ��ֵ
            int idx = raw[y * width + x];
            float mapped = (1 - dy) * (1 - dx) * ltm_curves[channel][idx] +
                dy * (1 - dx) * ltm_curves[channel][idx] +
                (1 - dy) * dx * ltm_curves[channel][idx] +
                dy * dx * ltm_curves[channel][idx];

            // Ӧ�� LTM ӳ����
            raw[y * width + x] = (U16)(cfg.ltm_strength * mapped + (1 - cfg.ltm_strength) * raw[y * width + x]);
        }
    }

    // �ͷ��ڴ�
    for (int i = 0; i < 4; ++i) {
        free(histograms[i]);
        free(ltm_curves[i]);
    }

    LOG("done.");
    return OK;
}