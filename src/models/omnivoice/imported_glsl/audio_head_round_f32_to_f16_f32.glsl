#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer Y { float y[]; };

float fp16_bits_to_fp32(uint h) {
    uint sign = (h & 0x8000u) << 16;
    uint exp = (h >> 10) & 0x1fu;
    uint mant = h & 0x03ffu;
    if (exp == 0u) {
        if (mant == 0u) {
            return uintBitsToFloat(sign);
        }
        float value = float(mant) * exp2(-24.0);
        return sign == 0u ? value : -value;
    }
    if (exp == 31u) {
        return uintBitsToFloat(sign | 0x7f800000u | (mant << 13));
    }
    uint fp32_exp = exp + 112u;
    return uintBitsToFloat(sign | (fp32_exp << 23) | (mant << 13));
}

float round_fp16_rne(float value) {
    uint bits = floatBitsToUint(value);
    uint sign = (bits >> 16) & 0x8000u;
    uint abs_bits = bits & 0x7fffffffu;
    uint exp = (abs_bits >> 23) & 0xffu;
    uint mant = abs_bits & 0x7fffffu;

    if (exp == 255u) {
        return value;
    }

    int half_exp = int(exp) - 127 + 15;
    if (half_exp >= 31) {
        return fp16_bits_to_fp32(sign | 0x7c00u);
    }
    if (half_exp <= 0) {
        if (half_exp < -10) {
            return fp16_bits_to_fp32(sign);
        }
        uint mantissa = mant | 0x800000u;
        uint shift = uint(14 - half_exp);
        uint half_mant = mantissa >> shift;
        uint round_bit = (mantissa >> (shift - 1u)) & 1u;
        uint sticky = mantissa & ((1u << (shift - 1u)) - 1u);
        if (round_bit != 0u && (sticky != 0u || (half_mant & 1u) != 0u)) {
            half_mant += 1u;
        }
        return fp16_bits_to_fp32(sign | half_mant);
    }

    uint half_mant = mant >> 13;
    uint round_bits = mant & 0x1fffu;
    if (round_bits > 0x1000u || (round_bits == 0x1000u && (half_mant & 1u) != 0u)) {
        half_mant += 1u;
        if (half_mant == 0x400u) {
            half_mant = 0u;
            half_exp += 1;
            if (half_exp >= 31) {
                return fp16_bits_to_fp32(sign | 0x7c00u);
            }
        }
    }

    return fp16_bits_to_fp32(sign | (uint(half_exp) << 10) | half_mant);
}

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    const uint n = uint(y.length());
    for (uint i = idx; i < n; i += total) {
        y[i] = round_fp16_rne(y[i]);
    }
}
