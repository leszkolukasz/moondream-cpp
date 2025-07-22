#include "moondreamwrapper.h"

MoondreamWrapper::MoondreamWrapper(QObject* parent)
    : QObject(parent) {}

bool MoondreamWrapper::isReady() const {
    return m_ready;
}

void MoondreamWrapper::load(const QString& modelPath) {
    QtConcurrent::run([=]() {
        try {
            m_moondream = std::make_unique<Moondream>(modelPath.toStdString());
            m_ready = true;
            emit readyChanged();
        } catch (const std::exception& e) {
            emit error(QString("Failed to load model: %1").arg(e.what()));
        }
    });
}

void MoondreamWrapper::caption(const QString& imagePath, const QString& mode) {
    if (!m_ready) {
        emit error("Model not loaded");
        return;
    }

    QtConcurrent::run([=]() {
        try {
            auto result = m_moondream->caption(
                imagePath.toStdString(),
                mode.toStdString(),
                100);
            emit captionResult(QString::fromStdString(result));
        } catch (const std::exception& e) {
            emit error(QString("Captioning failed: %1").arg(e.what()));
        }
    });
}
