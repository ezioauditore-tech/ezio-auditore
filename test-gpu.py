
import asyncio
async def function_asyc():
    for i in range(5):
        print("Hello, I'm Abhishek")
        print("GFG is Great")
    return 0 
# to run the above function we'll 
# use Event Loops these are low 
# level functions to run async functions
loop = asyncio.get_event_loop()
loop.run_until_complete(function_asyc())
loop.close()
print("HELLO WORLD")
# You can also use High Level functions Like:
# asyncio.run(function_asyc())