// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#define _USE_MATH_DEFINES // for M_PI

#include "llamafile/log.h"
#include "llamafile/llamafile.h"
#include "common.h"

// third-party utilities
// use your favorite implementations
// #define DR_WAV_IMPLEMENTATION // [jart] comment out
#include "dr_wav.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <cosmo.h>
#include <stdlib.h>
#include <unistd.h>

#include "stb/stb_vorbis.h"
#include "miniaudio.h"

#define MA_DATA_CONVERTER_STACK_BUFFER_SIZE 4096

static std::string delete_me;

static void on_exit(void) {
    if (!delete_me.empty()) {
        unlink(delete_me.c_str());
    }
}

static ma_result perform_audio_conversion(ma_decoder* pDecoder, ma_encoder* pEncoder) {
    ma_result rc = MA_SUCCESS;
    for (;;) {
        ma_uint8 pRawData[MA_DATA_CONVERTER_STACK_BUFFER_SIZE];
        ma_uint64 framesReadThisIteration;
        ma_uint64 framesToReadThisIteration;
        framesToReadThisIteration = sizeof(pRawData) / ma_get_bytes_per_frame(pDecoder->outputFormat, pDecoder->outputChannels);
        rc = ma_decoder_read_pcm_frames(pDecoder, pRawData, framesToReadThisIteration, &framesReadThisIteration);
        if (rc != MA_SUCCESS) {
            break;
        }
        ma_encoder_write_pcm_frames(pEncoder, pRawData, framesReadThisIteration, NULL);
        if (framesReadThisIteration < framesToReadThisIteration) {
            break;
        }
    }
    return rc;
}

// converts audio file to signed 16-bit 16000hz wav
static std::string convert_audio_file(const std::string & fname, bool stereo) {

    // create temporary filename
    std::string newpath;
    newpath = __get_tmpdir();
    newpath += "/whisperfile.";
    newpath += std::to_string(_rand64());
    newpath += ".wav";

    // create decoder
    ma_decoder_config decoderConfig =
            ma_decoder_config_init(ma_format_s16, 1 + stereo, COMMON_SAMPLE_RATE);
    decoderConfig.resampling.algorithm = ma_resample_algorithm_linear;
    decoderConfig.resampling.linear.lpfOrder = 8;

    // open input file
    ma_decoder decoder;
    ma_result rc = ma_decoder_init_file(fname.c_str(), &decoderConfig, &decoder);
    if (rc != MA_SUCCESS) {
        fprintf(stderr, "%s: failed to open audio file: %s (we support .wav, .mp3, .flac, and .ogg)\n",
                fname.c_str(), ma_result_description(rc));
        return "";
    }

    // create encoder
    ma_encoder encoder;
    ma_encoder_config encoderConfig = ma_encoder_config_init(
        ma_encoding_format_wav,
        decoder.outputFormat,
        decoder.outputChannels,
        decoder.outputSampleRate);
    rc = ma_encoder_init_file(newpath.c_str(), &encoderConfig, &encoder);
    if (rc != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        fprintf(stderr, "%s: failed to open output file: %s\n",
                newpath.c_str(), ma_result_description(rc));
        return "";
    }

    // perform the conversion
    rc = perform_audio_conversion(&decoder, &encoder);
    ma_encoder_uninit(&encoder);
    ma_decoder_uninit(&decoder);
    if (rc != MA_SUCCESS) {
        fprintf(stderr, "%s: failed to convert audio file: %s\n",
                fname.c_str(), ma_result_description(rc));
        return "";
    }

    // return new path
    delete_me = newpath;
    atexit(on_exit);
    return newpath;
}

#define TRY_CONVERSION                                                  \
    do {                                                                \
        if (did_conversion) {                                           \
            fprintf(stderr, "error: failed to open audio file\n");      \
            return false;                                               \
        }                                                               \
        std::string fname2 = convert_audio_file(fname, stereo);         \
        if (fname2.empty()) {                                           \
            return false;                                               \
        }                                                               \
        fname = fname2;                                                 \
        did_conversion = true;                                          \
        goto TryAgain;                                                  \
    } while (0)

bool read_wav(const std::string & fname_, std::vector<float>& pcmf32, std::vector<std::vector<float>>& pcmf32s, bool stereo) {
    drwav wav;
    std::vector<uint8_t> wav_data; // used for pipe input from stdin
    std::string fname = fname_;
    bool did_conversion = false;

TryAgain:
    if (fname == "-") {
        {
            #ifdef _WIN32
            _setmode(_fileno(stdin), _O_BINARY);
            #endif

            uint8_t buf[1024];
            while (true)
            {
                const size_t n = fread(buf, 1, sizeof(buf), stdin);
                if (n == 0) {
                    break;
                }
                wav_data.insert(wav_data.end(), buf, buf + n);
            }
        }

        if (drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr) == false) {
            fprintf(stderr, "error: failed to open WAV file from stdin\n");
            return false;
        }

        fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, wav_data.size());
    }
    else if (drwav_init_file(&wav, fname.c_str(), nullptr) == false) {
        tinylogf("%s: converting to wav...\n", fname.c_str());
        TRY_CONVERSION;
    }

    if (stereo && wav.channels < 2) {
        fprintf(stderr, "%s: audio file must be stereo for diarization\n", fname.c_str());
        drwav_uninit(&wav);
        return false;
    }

    if (wav.channels != 1 && wav.channels != 2) {
        tinylogf("%s: audio file has %d channels\n", fname.c_str(), wav.channels);
        drwav_uninit(&wav);
        TRY_CONVERSION;
    }

    if (stereo && wav.channels != 2) {
        tinylogf("%s: audio file has %d channels (we want diarization)\n", fname.c_str(), wav.channels);
        drwav_uninit(&wav);
        TRY_CONVERSION;
    }

    if (wav.sampleRate != COMMON_SAMPLE_RATE) {
        tinylogf("%s: audio file has %d sample rate\n", fname.c_str(), wav.sampleRate);
        drwav_uninit(&wav);
        TRY_CONVERSION;
    }

    if (wav.bitsPerSample != 16) {
        tinylogf("%s: audio file has %d bits per sample\n", fname.c_str(), wav.bitsPerSample);
        drwav_uninit(&wav);
        TRY_CONVERSION;
    }

    const uint64_t n = wav_data.empty() ? wav.totalPCMFrameCount : wav_data.size()/(wav.channels*wav.bitsPerSample/8);

    std::vector<int16_t> pcm16;
    pcm16.resize(n*wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);

    // convert to mono, float
    pcmf32.resize(n);
    if (wav.channels == 1) {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[i])/32768.0f;
        }
    } else {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;
        }
    }

    if (stereo) {
        // convert to stereo, float
        pcmf32s.resize(2);

        pcmf32s[0].resize(n);
        pcmf32s[1].resize(n);
        for (uint64_t i = 0; i < n; i++) {
            pcmf32s[0][i] = float(pcm16[2*i])/32768.0f;
            pcmf32s[1][i] = float(pcm16[2*i + 1])/32768.0f;
        }
    }

    return true;
}

void high_pass_filter(std::vector<float> & data, float cutoff, float sample_rate) {
    const float rc = 1.0f / (2.0f * M_PI * cutoff);
    const float dt = 1.0f / sample_rate;
    const float alpha = dt / (rc + dt);

    float y = data[0];

    for (size_t i = 1; i < data.size(); i++) {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

bool vad_simple(std::vector<float> & pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose) {
    const int n_samples      = pcmf32.size();
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        // not enough samples - assume no speech
        return false;
    }

    if (freq_thold > 0.0f) {
        high_pass_filter(pcmf32, freq_thold, sample_rate);
    }

    float energy_all  = 0.0f;
    float energy_last = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        energy_all += fabsf(pcmf32[i]);

        if (i >= n_samples - n_samples_last) {
            energy_last += fabsf(pcmf32[i]);
        }
    }

    energy_all  /= n_samples;
    energy_last /= n_samples_last;

    if (verbose) {
        tinylogf("%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n", __func__, energy_all, energy_last, vad_thold, freq_thold);
    }

    if (energy_last > vad_thold*energy_all) {
        return false;
    }

    return true;
}

float similarity(const std::string & s0, const std::string & s1) {
    const size_t len0 = s0.size() + 1;
    const size_t len1 = s1.size() + 1;

    std::vector<int> col(len1, 0);
    std::vector<int> prevCol(len1, 0);

    for (size_t i = 0; i < len1; i++) {
        prevCol[i] = i;
    }

    for (size_t i = 0; i < len0; i++) {
        col[0] = i;
        for (size_t j = 1; j < len1; j++) {
            col[j] = std::min(std::min(1 + col[j - 1], 1 + prevCol[j]), prevCol[j - 1] + (i > 0 && s0[i - 1] == s1[j - 1] ? 0 : 1));
        }
        col.swap(prevCol);
    }

    const float dist = prevCol[len1 - 1];

    return 1.0f - (dist / std::max(s0.size(), s1.size()));
}

bool sam_params_parse(int argc, char ** argv, sam_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-i" || arg == "--inp") {
            params.fname_inp = argv[++i];
        } else if (arg == "-o" || arg == "--out") {
            params.fname_out = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            sam_print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            // sam_print_usage(argc, argv, params); // [jart]
            exit(0);
        }
    }

    return true;
}

void sam_print_usage(int /*argc*/, char ** argv, const sam_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --inp FNAME\n");
    fprintf(stderr, "                        input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -o FNAME, --out FNAME\n");
    fprintf(stderr, "                        output file (default: %s)\n", params.fname_out.c_str());
    fprintf(stderr, "\n");
}

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

int timestamp_to_sample(int64_t t, int n_samples, int whisper_sample_rate) {
    return std::max(0, std::min((int) n_samples - 1, (int) ((t*whisper_sample_rate)/100)));
}

bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

bool speak_with_file(const std::string & command, const std::string & text, const std::string & path, int voice_id)
{
    std::ofstream speak_file(path.c_str());
    if (speak_file.fail()) {
        fprintf(stderr, "%s: failed to open speak_file\n", __func__);
        return false;
    } else {
        speak_file.write(text.c_str(), text.size());
        speak_file.close();
        int ret = system((command + " " + std::to_string(voice_id) + " " + path).c_str());
        if (ret != 0) {
            fprintf(stderr, "%s: failed to speak\n", __func__);
            return false;
        }
    }
    return true;
}
