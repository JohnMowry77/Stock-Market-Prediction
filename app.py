from flask import Flask, render_template, redirect, request, make_response
# import stock_data.py


# Create an instance of Flask
app=Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
	return render_template("index.html")


@app.route("/test", methods=["GET", "POST"])
def test():
	if request.method=="POST":
		startDate=request.form.get("startdate")
		endDate=request.form.get("enddate")
		print(startDate)
		print(endDate)
		# print(request.form.dict())
		# response=make_response()
		# return response
		# return redirect('/', test=test)

	return render_template("index.html")




if __name__=='__main__':
	app.run(debug=True)