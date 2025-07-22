#include <QGuiApplication>
#include <QQmlApplicationEngine>

#include "moondreamwrapper.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    qmlRegisterType<MoondreamWrapper>("moondream_mobile", 1, 0, "MoondreamWrapper");

    QQmlApplicationEngine engine;
    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreationFailed,
        &app,
        []() { QCoreApplication::exit(-1); },
        Qt::QueuedConnection);
    engine.loadFromModule("moondream_mobile", "Main");

    return app.exec();
}
