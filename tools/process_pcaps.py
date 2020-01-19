import os
import tldextract

print("[*] Processing PCAP")
extractor = 'tshark -r ../RAW_DATA/dnscat_C2.pcapng -T fields -e dns.qry.name -Y "dns.flags.response eq 0"'

pd = os.system(extractor)

outfile = open('test.csv', "a")
outfile.write('lld,\n')
for i in pd:
    e = tldextract.extract(i)
    if e.subdomain:
        print(e.subdomain)
        outfile.write(e.subdomain+'\n')
