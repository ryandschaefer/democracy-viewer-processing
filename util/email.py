from email.mime.text import MIMEText
from smtplib import SMTP_SSL
import os
from util.sql_connect import sql_connect
from util.sql_queries import get_user

def send_email(template: str, params: dict, subject: str, to: str, user: dict | None = None):
    # Set name based on provided fields
    if user is None:
        engine, meta = sql_connect()
        user = get_user(engine, meta, to)
    name = "{} {}".format(user["first_name"], user["last_name"])
    title = user["title"]
    suffix = user["suffix"]
    if title is not None:
        name = "{} {}".format(title, name)
    if suffix is not None:
        name = "{} {}".format(name, suffix)
    params["name"] = name
    
    # Fill in remaining parameters in email template
    with open("util/emails/{}.txt".format(template), "r") as file:
        full_text = ""
        for line in file.readlines():
            if len(line) > 0:
                text = "<p>{}</p>".format(line)
                text = text.replace("[[name]]", name)
                for key, val in params.items():
                    text = text.replace("[[{}]]".format(key), str(val))
                full_text += text    
                
        message = MIMEText(full_text, "html")

    # Set headers
    from_email = os.environ.get("FROM_EMAIL")
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = to
    
    # Send email
    server = SMTP_SSL(os.environ.get("EMAIL_SERVER"), int(os.environ.get("EMAIL_PORT")))
    # server.set_debuglevel(1)
    server.login(from_email, os.environ.get("EMAIL_PASSWORD"))
    server.sendmail(from_email, to, message.as_string())
    server.quit()