let currentButton = null;

function sendCommand(action) {

    fetch('/' + action)
        .then(res => res.json())
        .then(data => {

            updateStatus(data.action);
            highlightButton(action);
            updateIcon(action);

            document.getElementById("connectionStatus").className = "status-connection";
            document.getElementById("connectionStatus").innerText = "● CONNECTED";

        })
        .catch(() => {
            document.getElementById("connectionStatus").className = "status-connection offline";
            document.getElementById("connectionStatus").innerText = "● DISCONNECTED";
        });
}

function updateStatus(text) {
    const stat = document.getElementById("statusText");
    stat.innerText = text;
}

function updateIcon(action) {
    const icon = document.getElementById("directionIcon");

    switch (action) {
        case "forward":
            icon.innerText = "▲";
            break;
        case "backward":
            icon.innerText = "▼";
            break;
        case "left":
            icon.innerText = "◀";
            break;
        case "right":
            icon.innerText = "▶";
            break;
        default:
            icon.innerText = "⏹";
    }
}

function highlightButton(action) {

    if (currentButton) {
        currentButton.classList.remove("active");
    }

    const btn = document.getElementById("btn-" + action);
    if (btn) {
        btn.classList.add("active");
        currentButton = btn;
    }
}
                                            
function updateSpeed(value) {
    document.getElementById("speedValue").innerText = value + "%";
    let decimalSpeed = value / 100;

    // Corrected to match @app.route("/set_speed/<value>")
    fetch("/set_speed/" + decimalSpeed)
        .then(res => res.json())
        .then(data => console.log("Speed set to:", data.speed))
        .catch(err => console.error("Update failed:", err));
}

function exitLEDMode() {
    fetch('/led/mode/exit')
        .then(res => res.json())
        .then(data => {
            alert('Returning to motor control mode');
            window.location.href = '/';
        });
}
