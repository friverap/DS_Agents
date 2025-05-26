import React, { useState, useEffect } from 'react';
import { 
  Box, 
  FormControl, 
  FormLabel, 
  Select, 
  Button, 
  Flex, 
  Heading, 
  Text, 
  useToast,
  Card,
  CardBody,
  CardHeader,
  Badge,
  Divider,
  Input,
  InputGroup,
  InputRightElement,
  IconButton,
  Tooltip,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Tabs, 
  TabList, 
  TabPanels, 
  Tab, 
  TabPanel,
  SimpleGrid
} from '@chakra-ui/react';
import { modelsApi } from '../services/api';
import { InfoIcon, ViewIcon, ViewOffIcon, CheckIcon, WarningIcon } from '@chakra-ui/icons';

const ModelSelector = () => {
  const [providers, setProviders] = useState({});
  const [currentProvider, setCurrentProvider] = useState('');
  const [currentModel, setCurrentModel] = useState('');
  const [selectedProvider, setSelectedProvider] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [loading, setLoading] = useState(true);
  const [apiKeys, setApiKeys] = useState({});
  const [showApiKey, setShowApiKey] = useState({});
  const [currentApiStatus, setCurrentApiStatus] = useState({});
  const toast = useToast();

  const modelDescriptions = {
    'openai': {
      description: 'OpenAI models including GPT-4o and GPT-3.5 Turbo',
      website: 'https://openai.com',
      models: {
        'gpt-4o': 'Latest multimodal model with vision and voice capabilities',
        'gpt-4o-mini': 'Smaller, more affordable multimodal model',
        'gpt-4-turbo': 'Powerful and fast GPT-4 variant',
        'gpt-3.5-turbo': 'Cost-effective, general purpose model'
      }
    },
    'claude': {
      description: 'Anthropic\'s Claude models focus on being helpful, harmless, and honest',
      website: 'https://anthropic.com',
      models: {
        'claude-3-7-sonnet-20250219': 'Latest Claude model with advanced reasoning capabilities',
        'claude-3-5-sonnet-20241022': 'Powerful model balancing intelligence and efficiency',
        'claude-3-haiku': 'Fast and cost-effective for routine tasks'
      }
    },
    'google': {
      description: 'Google\'s Gemini models with strong reasoning capabilities',
      website: 'https://ai.google.dev',
      models: {
        'gemini-1.5-pro': 'High-performance model with long context window',
        'gemini-1.5-flash': 'Fast and efficient variant',
        'gemini-1.0-pro': 'Previous generation with solid performance'
      }
    },
    'mistral': {
      description: 'Mistral AI\'s efficient and powerful open models',
      website: 'https://mistral.ai',
      models: {
        'mistral-large': 'Most powerful Mistral model for complex tasks',
        'mistral-medium': 'Balanced capability and efficiency',
        'mistral-small': 'Cost-effective for most applications'
      }
    },
    'cohere': {
      description: 'Specialized models for natural language understanding and generation',
      website: 'https://cohere.ai',
      models: {
        'command': 'General purpose model for diverse tasks',
        'command-light': 'Efficient model for routine operations',
        'command-r': 'Enhanced reasoning capabilities',
        'command-r-plus': 'Most powerful Cohere model with extended reasoning'
      }
    },
    'groq': {
      description: 'Ultra-fast inference platform for LLMs',
      website: 'https://groq.com',
      models: {
        'llama3-70b-8192': 'High-performance Llama 3 model with 70B parameters',
        'llama3-8b-8192': 'Efficient Llama 3 model with 8B parameters',
        'mixtral-8x7b': 'Performant mixture-of-experts model'
      }
    },
    'ollama': {
      description: 'Local LLM inference for open models',
      website: 'https://ollama.ai',
      models: {
        'llama3': 'Meta\'s Llama 3 model running locally',
        'llama2': 'Meta\'s Llama 2 model running locally',
        'mistral': 'Mistral\'s open model running locally',
        'phi3': 'Microsoft\'s Phi-3 model running locally',
        'orca-mini': 'Lightweight model for basic tasks'
      }
    }
  };

  // Fetch available providers and current config on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Get available providers
        const providersResponse = await modelsApi.getProviders();
        setProviders(providersResponse.data);
        
        // Get current model config
        const currentModelResponse = await modelsApi.getCurrentModel();
        setCurrentProvider(currentModelResponse.data.provider);
        setCurrentModel(currentModelResponse.data.model);
        
        // Set initial selection to current configuration
        setSelectedProvider(currentModelResponse.data.provider);
        setSelectedModel(currentModelResponse.data.model);

        // Get API key status
        const apiStatusResponse = await modelsApi.getApiKeyStatus();
        setCurrentApiStatus(apiStatusResponse.data);
      } catch (error) {
        console.error('Error fetching model data:', error);
        toast({
          title: 'Error fetching model data',
          description: error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [toast]);

  // Handle provider change
  const handleProviderChange = (e) => {
    const provider = e.target.value;
    setSelectedProvider(provider);
    
    // Reset model selection to provider's default model
    if (providers[provider]) {
      setSelectedModel(providers[provider].default_model);
    }
  };

  // Handle model change
  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
  };

  // Save configuration
  const handleSave = async () => {
    try {
      setLoading(true);
      await modelsApi.configureModel(selectedProvider, selectedModel);
      
      // Update current configuration
      setCurrentProvider(selectedProvider);
      setCurrentModel(selectedModel);
      
      toast({
        title: 'Model configuration updated',
        description: `Using ${selectedProvider}/${selectedModel}`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error saving model configuration:', error);
      toast({
        title: 'Error saving model configuration',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle API key update
  const handleApiKeyChange = (provider, value) => {
    setApiKeys({
      ...apiKeys,
      [provider]: value
    });
  };

  // Toggle API key visibility
  const toggleApiKeyVisibility = (provider) => {
    setShowApiKey({
      ...showApiKey,
      [provider]: !showApiKey[provider]
    });
  };

  // Save API key
  const saveApiKey = async (provider) => {
    try {
      setLoading(true);
      await modelsApi.updateApiKey(provider, apiKeys[provider]);
      
      // Refresh API key status
      const apiStatusResponse = await modelsApi.getApiKeyStatus();
      setCurrentApiStatus(apiStatusResponse.data);
      
      toast({
        title: 'API Key Updated',
        description: `${provider.charAt(0).toUpperCase() + provider.slice(1)} API key has been updated`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      
      // Clear the displayed API key for security
      setApiKeys({
        ...apiKeys,
        [provider]: ''
      });
      setShowApiKey({
        ...showApiKey,
        [provider]: false
      });
    } catch (error) {
      console.error('Error updating API key:', error);
      toast({
        title: 'Error updating API key',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card mb={4}>
      <CardHeader>
        <Heading size="md">AI Model Configuration</Heading>
      </CardHeader>
      <CardBody>
        <Tabs variant="enclosed">
          <TabList>
            <Tab>Current Model</Tab>
            <Tab>API Keys</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
              <Box mb={4}>
                <Text fontWeight="bold" mb={2}>Active Configuration:</Text>
                <Flex alignItems="center" mb={2}>
                  <Badge colorScheme="green" mr={2} p={1}>
                    {currentProvider.charAt(0).toUpperCase() + currentProvider.slice(1)}
                  </Badge>
                  <Text>/</Text>
                  <Badge colorScheme="blue" ml={2} p={1}>
                    {currentModel}
                  </Badge>
                </Flex>
                {modelDescriptions[currentProvider] && (
                  <Text fontSize="sm" color="gray.600" mb={2}>
                    {modelDescriptions[currentProvider].description}
                  </Text>
                )}
              </Box>
              
              <Divider my={4} />
              
              <Heading size="sm" mb={3}>Change Model</Heading>
              <Flex direction="column" gap={4}>
                <FormControl>
                  <FormLabel>AI Provider</FormLabel>
                  <Select
                    value={selectedProvider}
                    onChange={handleProviderChange}
                    isDisabled={loading || Object.keys(providers).length === 0}
                  >
                    {Object.keys(providers).map((provider) => (
                      <option key={provider} value={provider}>
                        {provider.charAt(0).toUpperCase() + provider.slice(1)}
                        {currentApiStatus[provider] === false && " (API Key Missing)"}
                      </option>
                    ))}
                  </Select>
                </FormControl>
                
                <FormControl>
                  <FormLabel>Model</FormLabel>
                  <Select
                    value={selectedModel}
                    onChange={handleModelChange}
                    isDisabled={loading || !selectedProvider || !providers[selectedProvider]}
                  >
                    {selectedProvider && providers[selectedProvider]?.models.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </Select>
                  {selectedProvider && selectedModel && modelDescriptions[selectedProvider]?.models[selectedModel] && (
                    <Text fontSize="xs" color="gray.600" mt={1}>
                      {modelDescriptions[selectedProvider].models[selectedModel]}
                    </Text>
                  )}
                </FormControl>
                
                <Button
                  colorScheme="blue"
                  isLoading={loading}
                  onClick={handleSave}
                  isDisabled={
                    loading || 
                    !selectedProvider || 
                    !selectedModel || 
                    (selectedProvider === currentProvider && selectedModel === currentModel) ||
                    currentApiStatus[selectedProvider] === false
                  }
                  mt={2}
                >
                  Save Configuration
                </Button>
                
                {currentApiStatus[selectedProvider] === false && (
                  <Text color="red.500" fontSize="sm" mt={1}>
                    <WarningIcon mr={1} />
                    API key missing for {selectedProvider}. Please configure it in the API Keys tab.
                  </Text>
                )}
              </Flex>
            </TabPanel>
            
            <TabPanel>
              <Accordion allowMultiple>
                {Object.keys(providers).map((provider) => (
                  <AccordionItem key={provider}>
                    <h2>
                      <AccordionButton>
                        <Box flex="1" textAlign="left">
                          <Flex alignItems="center">
                            <Text fontWeight="medium">{provider.charAt(0).toUpperCase() + provider.slice(1)}</Text>
                            {currentApiStatus[provider] === true ? (
                              <Badge colorScheme="green" ml={2}>Configured</Badge>
                            ) : (
                              <Badge colorScheme="red" ml={2}>Not Configured</Badge>
                            )}
                          </Flex>
                        </Box>
                        <AccordionIcon />
                      </AccordionButton>
                    </h2>
                    <AccordionPanel pb={4}>
                      {provider !== 'ollama' ? (
                        <>
                          <Text fontSize="sm" mb={3}>
                            {modelDescriptions[provider]?.description}
                            {modelDescriptions[provider]?.website && (
                              <Tooltip label={`Visit ${modelDescriptions[provider].website}`}>
                                <IconButton
                                  as="a"
                                  href={modelDescriptions[provider].website}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  aria-label="Info"
                                  icon={<InfoIcon />}
                                  size="xs"
                                  ml={2}
                                  variant="ghost"
                                />
                              </Tooltip>
                            )}
                          </Text>
                          
                          <FormControl>
                            <FormLabel>API Key</FormLabel>
                            <InputGroup size="md">
                              <Input
                                pr="4.5rem"
                                type={showApiKey[provider] ? "text" : "password"}
                                placeholder={`Enter ${provider} API key`}
                                value={apiKeys[provider] || ''}
                                onChange={(e) => handleApiKeyChange(provider, e.target.value)}
                              />
                              <InputRightElement width="4.5rem">
                                <IconButton
                                  h="1.75rem"
                                  size="sm"
                                  onClick={() => toggleApiKeyVisibility(provider)}
                                  icon={showApiKey[provider] ? <ViewOffIcon /> : <ViewIcon />}
                                  variant="ghost"
                                />
                              </InputRightElement>
                            </InputGroup>
                          </FormControl>
                          
                          <Button
                            mt={4}
                            colorScheme="green"
                            isDisabled={!apiKeys[provider]}
                            onClick={() => saveApiKey(provider)}
                            size="sm"
                          >
                            Save API Key
                          </Button>
                        </>
                      ) : (
                        <Text fontSize="sm">
                          Ollama runs locally and does not require an API key. Make sure Ollama is installed and running on your machine.
                        </Text>
                      )}
                    </AccordionPanel>
                  </AccordionItem>
                ))}
              </Accordion>
            </TabPanel>
          </TabPanels>
        </Tabs>
        
        <Divider my={4} />
        
        <Heading size="sm" mb={3}>Available Models</Heading>
        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={4}>
          {Object.keys(providers).map((provider) => (
            <Box 
              key={provider} 
              p={3} 
              borderWidth="1px" 
              borderRadius="md" 
              borderColor={provider === currentProvider ? "blue.300" : "gray.200"}
              bg={provider === currentProvider ? "blue.50" : "white"}
            >
              <Flex justifyContent="space-between" alignItems="center" mb={2}>
                <Text fontWeight="bold">{provider.charAt(0).toUpperCase() + provider.slice(1)}</Text>
                {currentApiStatus[provider] === true ? (
                  <CheckIcon color="green.500" />
                ) : (
                  <WarningIcon color="orange.500" />
                )}
              </Flex>
              <Text fontSize="xs" color="gray.600" mb={2}>
                {modelDescriptions[provider]?.description?.substring(0, 80)}...
              </Text>
              <Text fontSize="xs" fontWeight="medium" color="gray.500">
                Available Models: {providers[provider]?.models.length}
              </Text>
            </Box>
          ))}
        </SimpleGrid>
      </CardBody>
    </Card>
  );
};

export default ModelSelector; 