param(
	[string]$filePath
)

$namespace = "http://schemas.microsoft.com/developer/msbuild/2003"

[XML]$config = Get-Content $filePath

Function Get-BmiNode([System.Xml.XmlDocument]$document) {
	$bmiNode = $document.createElement("AllProjectBMIsArePublic", $namespace)
	$bmiNode.InnerText = "True"
	return $bmiNode
}
$debugNode = $config.createElement("PropertyGroup", $namespace)
$releaseNode = $config.createElement("PropertyGroup", $namespace)



$debugNode.SetAttribute("Condition", "'`$(Configuration)|`$(Platform)'=='Debug|x64'")
$releaseNode.SetAttribute("Condition", "'`$(Configuration)|`$(Platform)'=='Release|x64'")

$bmiNode1 = Get-BmiNode $config
$bmiNode2 = Get-BmiNode $config 

$debugNode.AppendChild($bmiNode1)
$releaseNode.AppendChild($bmiNode2)
$config.project.AppendChild($debugNode)
$config.project.AppendChild($releaseNode)
$config.Save($filePath)