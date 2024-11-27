#include "demosaic.h"
#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))

RGB* demosaic_process(U16* raw, IMG_CONTEXT context, G_CONFIG cfg) {
    // ��ȡͼ��Ŀ�ߺ� Bayer Pattern
    U16 width = context.width;
    U16 height = context.height;
    BayerPattern pattern = (BayerPattern)cfg.pattern;
    ByteOrder order = (ByteOrder)cfg.order;
    const U8 bit_depth = cfg.bit;
    const int bit_shift = cfg.bit - 8;

    // ȷ�������Чֵ
    U16 max_val = (1 << bit_depth) - 1;

    // ȷ��ԭʼ���ݰ���С��˳����
    for (U32 i = 0; i < width * height; i++) {
        if (order == LITTLE_ENDIAN) {
            raw[i] = raw[i] & max_val; // ��ȡ�� bit_depth λ
        }
        else if (order == BIG_ENDIAN) {
            raw[i] = ((raw[i] & 0xFF) << 8 | (raw[i] >> 8)) & max_val;
        }
    }

    // ���� RGB ���ݵ��ڴ�
    RGB* rgb_data = (RGB*)malloc(width * height * sizeof(RGB));
    if (!rgb_data) {
        fprintf(stderr, "Memory allocation for RGB data failed.\n");
        return NULL;
    }

    // ���� Bayer Pattern ���в�ֵ����
    for (U16 y = 0; y < height; y++) {
        for (U16 x = 0; x < width; x++) {
            RGB pixel = { 0, 0, 0 };
            U32 val = 0;

            // �߽����߼�
            int left = (x > 0) ? x - 1 : x;
            int right = (x < width - 1) ? x + 1 : x;
            int top = (y > 0) ? y - 1 : y;
            int bottom = (y < height - 1) ? y + 1 : y;

            // ��ֵ����
            switch (pattern) {
            case RGGB:
                if ((y % 2 == 0) && (x % 2 == 0)) //R
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.r = clp_range(0, val, U8MAX);

                    val = ((U32)raw[y * width + left] + raw[y * width + right] +
                        raw[top * width + x] + raw[bottom * width + x]) >> (2 + bit_shift);
                    pixel.g = clp_range(0, val, U8MAX);

                    val = ((U32)raw[top * width + left] + raw[top * width + right] +
                        raw[bottom * width + left] + raw[bottom * width + right]) >> (2 + bit_shift);
                    pixel.b = clp_range(0, val, U8MAX);
                }
                else if ((y % 2 == 0) && (x % 2 == 1)) //GR
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.g = clp_range(0, val, U8MAX);

                    val = ((U32)raw[y * width + left] + raw[y * width + right]) >> (1 + bit_shift);
                    pixel.r = clp_range(0, val, U8MAX);

                    val = ((U32)raw[top * width + x] + raw[bottom * width + x]) >> (1 + bit_shift);
                    pixel.b = clp_range(0, val, U8MAX);
                }
                else if ((y % 2 == 1) && (x % 2 == 0)) //GB
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.g = clp_range(0, val, U8MAX);

                    val = ((U32)raw[top * width + x] + raw[bottom * width + x]) >> (1 + bit_shift);
                    pixel.r = clp_range(0, val, U8MAX);

                    val = ((U32)raw[y * width + left] + raw[y * width + right]) >> (1 + bit_shift);
                    pixel.b = clp_range(0, val, U8MAX); 
                }
                else //B
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.b = clp_range(0, val, U8MAX);

                    val = ((U32)raw[y * width + left] + raw[y * width + right] +
                        raw[top * width + x] + raw[bottom * width + x]) >> (2 + bit_shift);
                    pixel.g = clp_range(0, val, U8MAX);

                    val = ((U32)raw[top * width + left] + raw[top * width + right] +
                        raw[bottom * width + left] + raw[bottom * width + right]) >> (2 + bit_shift);
                    pixel.r = clp_range(0, val, U8MAX); 
                }
                break;
            case BGGR:
                if ((y % 2 == 0) && (x % 2 == 0)) //B
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.b = clp_range(0, val, U8MAX);

                    val = ((U32)raw[y * width + left] + raw[y * width + right] +
                        raw[top * width + x] + raw[bottom * width + x]) >> (2 + bit_shift);
                    pixel.g = clp_range(0, val, U8MAX);

                    val = ((U32)raw[top * width + left] + raw[top * width + right] +
                        raw[bottom * width + left] + raw[bottom * width + right]) >> (2 + bit_shift);
                    pixel.r = clp_range(0, val, U8MAX);
                }
                else if ((y % 2 == 0) && (x % 2 == 1)) //GB
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.g = clp_range(0, val, U8MAX);

                    val = ((U32)raw[top * width + x] + raw[bottom * width + x]) >> (1 + bit_shift);
                    pixel.r = clp_range(0, val, U8MAX);

                    val = ((U32)raw[y * width + left] + raw[y * width + right]) >> (1 + bit_shift);
                    pixel.b = clp_range(0, val, U8MAX);
                }
                else if ((y % 2 == 1) && (x % 2 == 0)) //GR
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.g = clp_range(0, val, U8MAX);

                    val = ((U32)raw[y * width + left] + raw[y * width + right]) >> (1 + bit_shift);
                    pixel.r = clp_range(0, val, U8MAX);

                    val = ((U32)raw[top * width + x] + raw[bottom * width + x]) >> (1 + bit_shift);
                    pixel.b = clp_range(0, val, U8MAX); 
                }
                else //R
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.r = clp_range(0, val, U8MAX);

                    val = ((U32)raw[y * width + left] + raw[y * width + right] +
                        raw[top * width + x] + raw[bottom * width + x]) >> (2 + bit_shift);
                    pixel.g = clp_range(0, val, U8MAX);

                    val = ((U32)raw[top * width + left] + raw[top * width + right] +
                        raw[bottom * width + left] + raw[bottom * width + right]) >> (2 + bit_shift);
                    pixel.b = clp_range(0, val, U8MAX);
                }
                break;
                // Other patterns (GRBG, GBRG, BGGR) can be implemented similarly
            default:
                fprintf(stderr, "Unsupported Bayer Pattern.\n");
                free(rgb_data);
                return NULL;
            }

            // ���浽 RGB ����
            rgb_data[y * width + x] = pixel;
        }
    }

    LOG("done.");
    
    return rgb_data;
}