from flask import Flask,render_template,request
from pickle import load

app = Flask(__name__)

f = open("model.pkl","rb")
model = load(f)
f.close()


@app.route("/",methods=["GET","POST"])
def home():
	if request.method == "POST":
		age = request.form.get("age")
		salary = request.form.get("salary")
		
		if not age or not salary:
			return render_template("home.html", msg="⚠️ Please fill in all fields!")
		try:
			age = int(age)
			salary = int(salary)
		except ValueError:
			return render_template("home.html", msg="⚠️ Age and Kilometers Driven must be valid numbers.")
		
		res = model.predict([[age,salary]])
		if res[0] == 1:
			val = "Purchased"
		else:
			val = "Not Purchased"
		return render_template("home.html",msg=val)
	else:
		return render_template("home.html")

if __name__ == "__main__":
	app.run(debug=True,use_reloader=True)
	