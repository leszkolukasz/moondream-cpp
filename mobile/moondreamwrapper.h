#pragma once

#include <QObject>
#include <QString>
#include <QImage>
#include <QFuture>
#include <QtConcurrent>
#include <moondream/moondream.h>

using namespace moondream;

class MoondreamWrapper : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool ready READ isReady NOTIFY readyChanged)
    Q_PROPERTY(bool running READ isRunning NOTIFY runningChanged)
public:
    explicit MoondreamWrapper(QObject* parent = nullptr);

    Q_INVOKABLE void load();
    Q_INVOKABLE void caption(const QString& imagePath, const QString& mode);

    bool isReady() const;
    bool isRunning() const;

signals:
    void readyChanged();
    void runningChanged();
    void captionResult(QString result);
    void partialCaptionResult(QString result);
    void error(QString message);

private:
    bool m_ready = false;
    bool m_running = false;
    std::unique_ptr<Moondream> m_moondream;
};
