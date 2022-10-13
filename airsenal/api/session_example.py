from flask import Flask, session

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = "blah"


@app.route("/visits-counter/")
def visits():
    session["visits"] = (session.get("visits") + 1) if "visits" in session else 1
    return f"Total visits: {session.get('visits')}"


@app.route("/delete-visits/")
def delete_visits():
    session.pop("visits", None)  # delete visits
    return "Visits deleted"


if __name__ == "__main__":
    app.run(host="0.0.0.0")
