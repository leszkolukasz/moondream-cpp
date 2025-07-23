#include "moondreamwrapper.h"
#include "modeldownloader.h"

MoondreamWrapper::MoondreamWrapper(QObject* parent)
    : QObject(parent) {}

bool MoondreamWrapper::isReady() const {
    return m_ready;
}

bool MoondreamWrapper::isRunning() const {
    return m_running;
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

    QFile sourceFile(imagePath);
    if (!sourceFile.open(QIODevice::ReadOnly)) {
        qDebug() << "Cannot open source image";
        emit error("Failed to open image file");
        return;
    }

    QString outDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QString tempImagePath = outDir + "/image.jpg";

    QFile destFile(tempImagePath);
    if (!destFile.open(QIODevice::WriteOnly)) {
        qDebug() << "Cannot create destination file";
        emit error("Failed to create temporary image file");
        return;
    }

    destFile.write(sourceFile.readAll());
    sourceFile.close();
    destFile.close();

    auto _ = QtConcurrent::run([=, this]() {
        try {
            m_running = true;
            emit runningChanged();

            auto cb = new std::function<void(const std::string&)>(
                [this](const std::string& partial) {
                    emit captionResult(QString::fromStdString(partial));
                }
            );

            auto result = m_moondream->caption(
                tempImagePath.toStdString(),
                mode.toStdString(),
                100,
                cb
                );

            delete cb;
            emit captionResult(QString::fromStdString(result));
        } catch (const std::exception& e) {
            emit error(QString("Captioning failed: %1").arg(e.what()));
        }

        m_running = false;
        emit runningChanged();
    });
}
