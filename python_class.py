import os


class car():
	
	# init method or constructor
	def __init__(self, model, color):
		self.model = model
		self.color = color
		
	def show(self):
		print("Model is", self.model )
		print("color is", self.color )
		
# both objects have different self which
# contain their attributes

audi = car("audi a4", "blue")
ferrari = car("ferrari 488", "green")

audi.show()	 # same output as car.show(audi)
ferrari.show() # same output as car.show(ferrari)

#note:we can also do like this

# print("Model for audi is ",audi.model)
# print("Colour for ferrari is ",ferrari.color)

#this happens because after assigning in the constructor the attributes are linked to that particular object
#here attributes(model,colour) are linked to objects(audi,ferrari) as we initialize them
# Behind the scene, in every instance method
# call, python sends the instances also with
# that method call like car.show(audi)
