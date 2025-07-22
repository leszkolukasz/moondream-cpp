#include "moondreamwrapper.h"
#include "modeldownloader.h"

MoondreamWrapper::MoondreamWrapper(QObject* parent)
    : QObject(parent) {}

bool MoondreamWrapper::isReady() const {
    return m_ready;
}

void MoondreamWrapper::load() {
    QStringList files {
        "https://huggingface.co/whistleroosh/moondream-0.5B/resolve/main/config.json",
        "https://huggingface.co/whistleroosh/moondream-0.5B/resolve/main/coord_decoder.onnx",
        "https://huggingface.co/whistleroosh/moondream-0.5B/resolve/main/coord_encoder.onnx",
        "https://huggingface.co/whistleroosh/moondream-0.5B/resolve/main/initial_kv_cache.npy",
        "https://huggingface.co/whistleroosh/moondream-0.5B/resolve/main/size_decoder.onnx",
        "https://huggingface.co/whistleroosh/moondream-0.5B/resolve/main/size_encoder.onnx",
        "https://huggingface.co/whistleroosh/moondream-0.5B/resolve/main/text_decoder.onnx",
        "https://huggingface.co/whistleroosh/moondream-0.5B/resolve/main/text_encoder.onnx",
        "https://huggingface.co/whistleroosh/moondream-0.5B/resolve/main/tokenizer.json",
        "https://huggingface.co/whistleroosh/moondream-0.5B/resolve/main/vision_encoder.onnx",
        "https://huggingface.co/whistleroosh/moondream-0.5B/resolve/main/vision_projection.onnx",
        "https://raw.githubusercontent.com/leszkolukasz/moondream-cpp/master/frieren.jpg"
    };

    QString outDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/models";

    ModelDownloader* downloader = new ModelDownloader(this);
    connect(downloader, &ModelDownloader::allDownloadsFinished, this, [this, outDir]() {
        auto _ = QtConcurrent::run([=, this]() {
            try {
                m_moondream = std::make_unique<Moondream>(outDir.toStdString());
                m_ready = true;
                emit readyChanged();
            } catch (const std::exception& e) {
                emit error(QString("Failed to load model: %1").arg(e.what()));
            }
        });
    });
    connect(downloader, &ModelDownloader::downloadError, this, [](const QString& msg) {
        qWarning() << "Download error:" << msg;
    });
    downloader->downloadFiles(files, outDir);
}


void MoondreamWrapper::caption(const QString& imagePath, const QString& mode) {
    if (!m_ready) {
        emit error("Model not loaded");
        return;
    }

    qDebug() << imagePath;

    QString outDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/models";

    auto _ = QtConcurrent::run([=, this]() {
        try {
            auto result = m_moondream->caption(
                // imagePath.toStdString(),
                outDir.toStdString() + "/frieren.jpg",
                mode.toStdString(),
                100);
            emit captionResult(QString::fromStdString(result));
        } catch (const std::exception& e) {
            emit error(QString("Captioning failed: %1").arg(e.what()));
        }
    });
}
