#include "modeldownloader.h"
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QDir>
#include <QFile>
#include <QStandardPaths>
#include <QDebug>

ModelDownloader::ModelDownloader(QObject* parent)
    : QObject(parent), manager(new QNetworkAccessManager(this))
{
    connect(manager, &QNetworkAccessManager::finished, this, &ModelDownloader::onFinished);
}

void ModelDownloader::downloadFiles(const QStringList& urlsList, const QString& outDirPath)
{
    urls = urlsList;
    outDir = outDirPath;
    currentIndex = 0;

    QDir dir(outDir);
    if (!dir.exists()) {
        dir.mkpath(".");
    }

    downloadNext();
}

void ModelDownloader::downloadNext()
{
    if (currentIndex >= urls.size()) {
        emit allDownloadsFinished();
        return;
    }

    if (reply) {
        reply->deleteLater();
        reply = nullptr;
    }

    QUrl url(urls[currentIndex]);
    QString fileName = url.path().section('/', -1);
    QString filePath = outDir + "/" + fileName;

    QFile file(filePath);
    if (file.exists()) {
        qDebug() << "File already exists, skipping:" << fileName;
        currentIndex++;
        downloadNext();
        return;
    }

    qDebug() << "Downloading:" << fileName;
    QNetworkRequest request(url);
    reply = manager->get(request);
    connect(reply, &QNetworkReply::downloadProgress, this, &ModelDownloader::onProgress);
}


void ModelDownloader::onFinished()
{
    if (!reply)
        return;

    if (reply->error() != QNetworkReply::NoError) {
        emit downloadError(QString("Failed to download %1: %2").arg(urls[currentIndex], reply->errorString()));
        reply->deleteLater();
        reply = nullptr;
        return;
    }

    QByteArray data = reply->readAll();
    QString fileName = urls[currentIndex].section('/', -1).section('?', 0, 0);
    QString filePath = outDir + "/" + fileName;

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        emit downloadError(QString("Failed to save file %1").arg(filePath));
        reply->deleteLater();
        reply = nullptr;
        return;
    }

    file.write(data);
    file.close();

    currentIndex++;
    reply->deleteLater();
    reply = nullptr;

    downloadNext();
}

void ModelDownloader::onProgress(qint64 bytesReceived, qint64 bytesTotal)
{
    Q_UNUSED(bytesReceived)
    Q_UNUSED(bytesTotal)
    // Optional: implement progress reporting here
}
