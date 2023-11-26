import smtplib

def send():
    try:
        x=smtplib.SMTP('smtp.gmail.com',587)
        x.starttls()
        x.login("noparanoyaaa@gmail.com", "")
        subject="Testing"
        body_text="Testing success"
        message="Subject: {}{}".format(subject,body_text)
        x.sendmail("noparanoyaaa@gmail.com","outofthebag15@gmail.com",message)
        print("Success")
    except Exception as exception:
        print(exception)
        print("Failure")

if __name__ == "__main__":
    send()
