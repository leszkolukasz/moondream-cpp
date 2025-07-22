#pragma once

#include <QObject>
#include <QStringList>

class QNetworkAccessManager;
class QNetworkReply;

class ModelDownloader : public QObject {
    Q_OBJECT
public:
    explicit ModelDownloader(QObject* parent = nullptr);

    void downloadFiles(const QStringList& urls, const QString& outDirPath);

signals:
    void allDownloadsFinished();
    void downloadError(const QString& message);

private slots:
    void downloadNext();
    void onFinished();
    void onProgress(qint64 bytesReceived, qint64 bytesTotal);

private:
    QNetworkAccessManager* manager;
    QNetworkReply* reply = nullptr;
    QStringList urls;
    QString outDir;
    int currentIndex = 0;
};
