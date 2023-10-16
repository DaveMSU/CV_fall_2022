from app import tensorboard_app


HOST_IP_ADRESS="192.168.0.239"
PORT_ID=5001


def main():
    tensorboard_app.run(
        host=HOST_IP_ADRESS,
        port=PORT_ID,
        debug=True
    )


if __name__ == "__main__":
    main()

