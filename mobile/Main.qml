import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs
import QtMultimedia 5.15

import moondream_mobile 1.0

ApplicationWindow {
    visible: true
    width: 400
    height: 500
    title: "Moondream Captioning"

    MoondreamWrapper {
        id: moondream
        onReadyChanged: console.log("Model ready:", ready)
        onPartialCaptionResult: (result) => resultText.text = result
        onCaptionResult: (result) => resultText.text = result
        onError: console.log("Error:", message)
    }

    Column {
        anchors.centerIn: parent
        spacing: 16
        width: parent.width * 0.8
        anchors.horizontalCenter: parent.horizontalCenter

        Button {
            text: "Load Model"
            width: parent.width
            onClicked: moondream.load()
            enabled: !moondream.running && !moondream.ready
        }

        Button {
            text: "Pick Image"
            width: parent.width
            onClicked: fileDialog.open()
            enabled: !moondream.running
        }

        Image {
            id: selectedImage
            width: 300
            height: 300
            fillMode: Image.PreserveAspectFit
            visible: source !== ""
            clip: true
            anchors.horizontalCenter: parent.horizontalCenter
        }

        Button {
            text: "Caption Image"
            width: parent.width
            enabled: moondream.ready && selectedImage.source.toString() !== "" && !moondream.running
            onClicked: moondream.caption(selectedImage.source.toString(), "short")
        }

        Text {
            id: resultText
            text: ""
            wrapMode: Text.Wrap
            width: parent.width
            horizontalAlignment: Text.AlignHCenter
        }
    }

    FileDialog {
        id: fileDialog
        title: "Select an image"
        nameFilters: ["Images (*.png *.jpg *.jpeg)"]
        onAccepted: selectedImage.source = fileDialog.selectedFile
    }
}

