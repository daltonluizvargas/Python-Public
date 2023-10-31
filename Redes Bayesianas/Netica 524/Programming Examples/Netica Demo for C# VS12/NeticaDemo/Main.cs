
/*
"Demo" program for Netica-COM in C#

Works with  Netica 5.19 or later.

It is best to first get this example project working.
Run the latest version of Netica Application (Netica.exe) as Administrator (e.g. by right-clicking
   Netica.exe and choosing "Run as Administrator"), and then exit, to register it as the COM server.
Then build this project (Build->Build Solution), and run it (Debug->Start Debugging).
If it reports the probability of Tuberculosis is 0.0104, then 0.09241, .3377, and .05 your Netica
   installation and C# project appear to all be in good order.

Now you can replace the code below with your own.
Or, to add Netica to a different project:
   From that project, choose Project->Add Reference, then the "COM" tab,
   then double-click "Netica 5.19 Object Library" (or similar).

For documentation on Netica's objects and functions:
   Add Netica to your project, as described above.
   Choose View->Object Browser  (or View->Other Windows->Object Browser)  or  click on the Views 
      multi-purpose tool button down-arrow and choose "Object Browser".
   In the left pane of the Object Browser, one of the top level entries will be "Interop.Netica".  
      You can browse that, but it won't be as good as browsing the Netica library directly, 
      because it won't have a description of each function, and it may not even list the functions at all.
      If there is no entry at the top level for the library named simply "Netica" (with the books icon), 
      choose "Edit Custom Component Set..." from the "Browse" menu on the toolbar of the Object Browser. 
      Choose the COM tab, select the Netica library from  the list, click "Add" or "Select" and then "OK".  
      Now the books icon for the Netica library should appear, and you can browse it.
   Choose a Netica object in the left pane, then click on a member function or property in the top
      right pane to view its short description in the bottom pane.
   For more information on it, find its Netica-C equivalent name at the end of its short description, 
      and look up the function in the Netica-C manual or online at http://www.norsys.com/onLineAPIManual/index.html .
   If you are getting the wrong version of Netica's objects, then you need to run the correct version of 
      Netica.exe first (by right clicking on the Netica.exe icon and choosing "Run as Administrator"), 
      to register its COM definition.
*/

using System;
using Netica;

namespace NeticaDemo
{
	class Program
	{
		static void Main(string[] args)
		{
			Console.WriteLine("Welcome to Netica API for C# !");
			try
			{
				Netica.Application app = new Netica.Application();
				Console.WriteLine("Using Netica version " + app.VersionString);
				app.Visible = true;

				string net_file_name = AppDomain.CurrentDomain.BaseDirectory + "..\\..\\..\\ChestClinic.dne";

				Streamer file = app.NewStream(net_file_name, null);
				BNet net = app.ReadBNet(file, "");
				net.Compile();
				BNode TB = net.Node("Tuberculosis");
				double belief = TB.GetBelief("present");
				Console.WriteLine("The probability of tuberculosis is " + belief.ToString("G4"));

				BNode XRay = net.Node("XRay");
				XRay.EnterFinding("abnormal");
				belief = TB.GetBelief("present");
				Console.WriteLine("Given an abnormal X-Ray, the probability of tuberculosis is " + belief.ToString("G4"));

				net.Node("VisitAsia").EnterFinding("visit");
				belief = TB.GetBelief("present");
				Console.WriteLine("Given abnormal X-Ray and visit to Asia, the probability of TB is " + belief.ToString("G4"));

				net.Node("Cancer").EnterFinding("present");
				belief = TB.GetBelief("present");
				Console.WriteLine("Given abnormal X-Ray, Asia visit, and lung cancer, the probability of TB is " + belief.ToString("G4"));

				net.Delete();
				// if (!app.UserControl) app.Quit();			// In most applications you would want to do this, but not for a demo
			}
			catch (System.Exception e)							// You could make this System.Runtime.InteropServices.COMException
			{
				Console.WriteLine(e);
			}
			Console.WriteLine("Press <enter> to quit.");
			Console.ReadLine();
		}
	}
}