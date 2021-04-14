#include "SeluPlugin.h"
#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::PluginType;
using nvinfer1::plugin::SeluPluginCreator;
using nvinfer1::plugin::Selu;

static const char* SELU_PLUGIN_VERSION{"1"};
static const char* SELU_PLUGIN_NAME{"Selu_TRT"};
PluginFieldCollection SeluPluginCreator::mFC{};
std::vector<PluginField> SeluPluginCreator::mPluginAttributes;

// LeakyReLU {{{
Selu::Selu(float lambd)
    : mLambd(lambd)
    , mBatchDim(1)
{
}

Selu::Selu(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    mLambd = read<float>(d);
    mBatchDim = read<int>(d);
    ASSERT(d == a + length);
}

int Selu::getNbOutputs() const
{
    return 1;
}

Dims Selu::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return inputs[0];
}

int Selu::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = seluInference(stream, mBatchDim * batchSize, mLambd, inputData, outputData);
    ASSERT(status == STATUS_SUCCESS);
    return status;
}

size_t Selu::getSerializationSize() const
{
    // mLambd, mBatchDim
    return sizeof(float) + sizeof(int);
}

void Selu::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mLambd);
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}

void Selu::configureWithFormat(
    const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int)
{
    ASSERT(type == DataType::kFLOAT && format == PluginFormat::kNCHW);
//    ASSERT(mBatchDim == 1);
    ASSERT(nbOutputs == 1);
    for (int i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

bool Selu::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

int Selu::initialize()
{
    return 0;
}

void Selu::terminate() {}

size_t Selu::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

const char* Selu::getPluginType() const
{
    return SELU_PLUGIN_NAME;
}

const char* Selu::getPluginVersion() const
{
    return SELU_PLUGIN_VERSION;
}

void Selu::destroy()
{
    delete this;
}

IPluginV2* Selu::clone() const
{
    IPluginV2* plugin = new Selu(mLambd);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

SeluPluginCreator::SeluPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("lambd", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SeluPluginCreator::getPluginName() const
{
    return SELU_PLUGIN_NAME;
}

const char* SeluPluginCreator::getPluginVersion() const
{
    return SELU_PLUGIN_VERSION;
}

const PluginFieldCollection* SeluPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* SeluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    ASSERT(fc->nbFields == 1);
    ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
    lambd = *(static_cast<const float*>(fields[0].data));

    return new Selu(lambd);
}

IPluginV2* SeluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call SeluPlugin::destroy()
    return new Selu(serialData, serialLength);
}
// LeakReLU }}}
